<script lang="ts">
	let actual_user: string;
	import Menu from '@smui/menu';
	import List, { Item, Separator, Text } from '@smui/list';
	import Button from '@smui/button';
	export let users;
	let menu;
	let clicked = 'nothing yet';
	console.log(users)
</script>

	<div style="min-width: 100px;">
		<Button on:click={() => menu.setOpen(true)}>...</Button>
		<Menu bind:this={menu}>
		  <List>
			<Item on:SMUI:action={() => (clicked = 'Cut')}>
			  <Text>Cut</Text>
			</Item>
			<Item on:SMUI:action={() => (clicked = 'Copy')}>
			  <Text>Copy</Text>
			</Item>
			<Item on:SMUI:action={() => (clicked = 'Paste')}>
			  <Text>Paste</Text>
			</Item>
			<Separator />
			<Item on:SMUI:action={() => (clicked = 'Delete')}>
			  <Text>Delete</Text>
			</Item>
		  </List>
		</Menu>
	  </div>
