<script lang="ts">
	export let name: string;

	import Grid from './Grid.svelte';
	import { onMount } from 'svelte';
	import { actual_user, users, beers, actual_beer} from './store.js';
	let user_id;
	actual_user.subscribe(value => {
		user_id = value;
		changeUser(user_id.id)
	});
	async function changeUser(id){
		{
		/////Get beer info ////
		if( Number.isInteger(id)){
			const beers_req = await fetch(`http://127.0.0.1:8000/get_rec/?user_id=${id}&k=10`,
			{headers: {'Content-Type': 'application/json',},})
			.then(
				res => res.json())
			.then(
				res=> {
					console.log(res)
					let act= res.recommended.data[0]
					beers.set(res);
					actual_beer.set(act)
				});
			
			;
		}
		}
		
	}
	onMount(async () =>{
	/// Get users ///
	const res = await fetch(`http://127.0.0.1:8000/getusers/`,{headers: {'Content-Type': 'application/json',},})
		.then(
			res => res.json())
		.then(
			res=> {
				let j = JSON.parse(res)
				actual_user.set(j[0]);
				users.set(j)}
			);
		console.log(user_id)
		//await changeUser(user_id.id)
	})

				
</script>
<Grid></Grid>

<style>
	
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
		background: black;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>