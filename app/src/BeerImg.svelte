<script lang="ts">
	let img = "./tmp.png"
	
</script>

	<div>
		<img class="beer_img" src={img}>
		<center><div class="beer_desc">Ocena 4.0</div></center>
	</div>

<style>
	.beer_img{
		width: 100%;
		
	}
	.beer_desc{
		position: relative;
		bottom:25px;
		color:white;
		background: rgba(0, 0, 0, 0.3);
	}
</style>